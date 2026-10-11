import { auth } from "@clerk/nextjs/server";

import { Detector } from "@/components/detector";

export default async function Home() {
  const { userId, redirectToSignIn } = await auth();
  if (!userId) return redirectToSignIn();
  return <Detector />;
}
